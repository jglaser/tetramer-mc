//! Standalone, bounded two-sphere equilibrium control, not a production kernel.
//!
//! The virtual tree frame is NOT a third physical sphere. Root confinement is
//! independent of root-child separation; no wall applies to the child. Hence
//! the physical radial marginal is d^2 exp[z V_overlap(d)], with uniform root
//! position and independent Haar orientations. The d^2 is NOT an MH factor.
//! Each attempt is one joint learned tree map followed by one MH decision,
//! then two exact sphere-orientation Gibbs refreshes at fixed WORLD centers.
//! The analytic and Poisson arms use alternative physical factors, never both.
//! All attempts, including burn-in, nulls and rejections, are written to JSONL.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{RngExt, SeedableRng, distr::Open01, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    f64::consts::PI,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};
use tetramer_mc::{
    defensive_dimer_proposal::{DefensiveDimerOutcome, DefensiveDimerProposal},
    depletion::{GateOptions, GateResult},
    dimer_tree_proposal::{DimerTreeOutcome, DimerTreeProposal},
    docking::{DockingMethod, DockingProposal},
    flexible_subset::FlexibleSubset,
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, norm, rotation, sub},
    proposal::FrozenRelativePoseProposal,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
    singleton_path::{SingletonPath, SingletonPathResult},
};

const FRAME: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};
const RNG_DOMAIN: &[u8] = b"dimer-tree-equilibrium-v1";

#[derive(Parser)]
#[command(about = "Bounded two-sphere validation of joint tree maps and the flexible bath gate")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    /// Must not exist. This control has no continuation or retry mode.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum Arm {
    Analytic,
    Poisson,
    SingletonPath,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ProposalKind {
    #[default]
    Tree,
    DefensiveIndependent,
}

fn default_uniform_probability() -> f64 {
    0.5
}
fn default_uniform_half_width() -> f64 {
    2.5
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Initialization {
    Contact,
    Dispersed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Chain {
    id: String,
    arm: Arm,
    initialization: Initialization,
    seed: u64,
    #[serde(default)]
    proposal_kind: ProposalKind,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Gate {
    max_cells: usize,
    max_depth: u32,
    min_width: f64,
}

impl Gate {
    fn options(self) -> GateOptions {
        GateOptions {
            max_cells: self.max_cells,
            max_depth: self.max_depth,
            min_width: self.min_width,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: u32,
    core_radius: f64,
    exclusion_radius: f64,
    activity: f64,
    auxiliary_intensity: f64,
    root_radius: f64,
    min_separation: f64,
    max_separation: f64,
    rho: f64,
    #[serde(default = "default_uniform_probability")]
    uniform_probability: f64,
    #[serde(default = "default_uniform_half_width")]
    uniform_half_width: f64,
    gate: Gate,
    burn_in: usize,
    production_steps: usize,
    chains: Vec<Chain>,
}

impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(self.schema == 1, "unsupported control schema");
        ensure!(
            [
                self.core_radius,
                self.exclusion_radius,
                self.auxiliary_intensity,
                self.root_radius,
                self.min_separation,
                self.max_separation
            ]
            .iter()
            .all(|x| x.is_finite() && *x > 0.),
            "radii, shell bounds and auxiliary intensity must be positive and finite"
        );
        ensure!(
            self.exclusion_radius >= self.core_radius,
            "negative depletant radius"
        );
        ensure!(
            self.min_separation >= 2. * self.core_radius
                && self.max_separation > self.min_separation,
            "invalid hard-core shell"
        );
        ensure!(
            self.activity.is_finite()
                && self.activity >= 0.
                && (self.auxiliary_intensity + self.activity).is_finite(),
            "invalid activity"
        );
        ensure!(
            self.rho.is_finite() && (0. ..=1.).contains(&self.rho),
            "invalid map correlation"
        );
        self.gate.options().validate()?;
        ensure!(
            self.uniform_probability.is_finite()
                && self.uniform_probability > 0.
                && self.uniform_probability < 1.
                && self.uniform_half_width.is_finite()
                && self.uniform_half_width > 0.,
            "invalid defensive proposal settings"
        );
        ensure!(
            self.production_steps > 0 && !self.chains.is_empty(),
            "empty allocation"
        );
        let attempts = self
            .burn_in
            .checked_add(self.production_steps)
            .context("attempt count overflow")?;
        attempts
            .checked_mul(self.chains.len())
            .context("total attempt count overflow")?;
        let mut ids = BTreeSet::new();
        let mut seeds = BTreeSet::new();
        for chain in &self.chains {
            ensure!(
                !chain.id.is_empty()
                    && chain
                        .id
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
                "invalid chain id"
            );
            ensure!(ids.insert(&chain.id), "duplicate chain id");
            ensure!(seeds.insert(chain.seed), "duplicate chain seed");
            ensure!(
                in_domain(self, initial_state(chain.initialization)),
                "fixed initialization is outside requested domain"
            );
        }
        Ok(())
    }
}

fn initial_state(initialization: Initialization) -> [Pose; 2] {
    // Fixed, predeclared initializations of this toy, not randomized retries.
    let (root, child) = match initialization {
        Initialization::Contact => ([0.; 3], [0.45, 0., 0.]),
        Initialization::Dispersed => ([1.5, 0., 0.], [3.9, 0., 0.]),
    };
    [
        Pose {
            position: root,
            ..FRAME
        },
        Pose {
            position: child,
            ..FRAME
        },
    ]
}

fn separation(state: [Pose; 2]) -> f64 {
    norm(sub(state[1].position, state[0].position))
}

fn in_domain(config: &Config, state: [Pose; 2]) -> bool {
    let d = separation(state);
    norm(state[0].position) <= config.root_radius
        && d >= config.min_separation
        && d <= config.max_separation
}

fn overlap(radius: f64, d: f64) -> f64 {
    if d >= 2. * radius {
        0.
    } else {
        PI * (4. * radius + d) * (2. * radius - d).powi(2) / 12.
    }
}

fn sphere(radius: f64) -> Shape {
    Shape {
        name: "dimer-tree-equilibrium hard sphere".into(),
        volume: 4. * PI * radius.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius,
        }],
    }
}

fn atlas(shape_sha: &str) -> Value {
    let quaternions = [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ];
    let covariance: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
    json!({"coordinate_convention":"anchor-body-relative", "shape_sha256":shape_sha,
        "angular_length":1., "weights":vec![0.25;4],
        "anchors":quaternions.map(|q|json!({"position":vec![0.;3],"rotation":rotation(q)})),
        "means":vec![[0.;6];4], "covariances":vec![covariance;4]})
}

fn stream_seed(seed: u64, label: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(RNG_DOMAIN);
    hasher.update(seed.to_le_bytes());
    hasher.update(label.as_bytes());
    hasher.finalize().into()
}

fn stream(seed: u64, label: &str) -> StdRng {
    StdRng::from_seed(stream_seed(seed, label))
}

fn haar(rng: &mut StdRng) -> Result<[f64; 4]> {
    let mut q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    ensure!(n > 0. && n.is_finite(), "invalid orientation refresh");
    for value in &mut q {
        *value /= n;
    }
    Ok(q)
}

#[derive(Serialize)]
struct Attempt {
    chain_id: String,
    step: usize,
    burn_in: bool,
    old_state: [Pose; 2],
    proposed_state: Option<[Pose; 2]>,
    /// Immediately after joint-map MH, before exact orientation refresh.
    mh_state: [Pose; 2],
    /// After two independent Haar refreshes, with world centers fixed.
    state: [Pose; 2],
    old_d: f64,
    proposed_d: Option<f64>,
    d: f64,
    proposal: Option<DimerTreeOutcome>,
    defensive_proposal: Option<DefensiveDimerOutcome>,
    proposal_null: bool,
    domain_valid: bool,
    hard_valid: bool,
    q_correction: Option<f64>,
    /// Diagnostic in either arm; NEVER added to the Poisson gate factor.
    analytic_log_weight: Option<f64>,
    gate: Option<GateResult>,
    path_gate: Option<SingletonPathResult>,
    log_ratio: Option<f64>,
    log_uniform: f64,
    accepted: bool,
    orientation_refresh_count: usize,
    attempt_cpu_seconds: f64,
    cpu_seconds: f64,
    error: Option<String>,
}

#[derive(Default, Serialize)]
struct Counts {
    attempts: usize,
    accepted: usize,
    proposal_null: usize,
    domain_rejected: usize,
    hard_rejected: usize,
    gate_calls: usize,
    raw_points: u64,
    retained_points: u64,
    orientation_refresh_count: usize,
    errors: usize,
}

fn write_line(writer: &mut BufWriter<File>, row: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, row)?;
    writer.write_all(b"\n")?;
    // Every completed attempt is visible before starting the next one.
    writer.flush()?;
    Ok(())
}

fn run_chain(
    config: &Config,
    chain: &Chain,
    tree: &SphereTree,
    docking: &DockingProposal,
    writer: &mut BufWriter<File>,
) -> Result<Value> {
    let proposal = DimerTreeProposal::new(docking)?;
    let defensive = DefensiveDimerProposal::new(
        docking,
        config.uniform_half_width,
        config.uniform_probability,
    )?;
    let mut proposal_rng = stream(chain.seed, "proposal");
    let mut gate_rng = stream(chain.seed, "gate");
    let mut mh_rng = stream(chain.seed, "mh");
    let mut refresh_rng = stream(chain.seed, "orientation_refresh");
    let mut state = initial_state(chain.initialization);
    let initial = FlexibleSubset::new(
        tree,
        &state,
        &[0, 1],
        &state,
        config.exclusion_radius - config.core_radius,
    )?;
    ensure!(
        initial.hard_valid(None, [0.; 3]),
        "hard-invalid initial state"
    );
    let mut counts = Counts::default();
    let started = cpu_seconds();
    for step in 0..config.burn_in + config.production_steps {
        let attempt_started = cpu_seconds();
        let uniform: f64 = mh_rng.sample(Open01);
        let mut row = Attempt {
            chain_id: chain.id.clone(),
            step,
            burn_in: step < config.burn_in,
            old_state: state,
            proposed_state: None,
            mh_state: state,
            state,
            old_d: separation(state),
            proposed_d: None,
            d: separation(state),
            proposal: None,
            defensive_proposal: None,
            proposal_null: true,
            domain_valid: false,
            hard_valid: false,
            q_correction: None,
            analytic_log_weight: None,
            gate: None,
            path_gate: None,
            log_ratio: None,
            log_uniform: uniform.ln(),
            accepted: false,
            orientation_refresh_count: 0,
            attempt_cpu_seconds: 0.,
            cpu_seconds: 0.,
            error: None,
        };
        let result = (|| -> Result<()> {
            let candidate = match chain.proposal_kind {
                ProposalKind::Tree => {
                    row.proposal =
                        Some(proposal.propose(&mut proposal_rng, FRAME, state[0], state[1])?);
                    row.proposal
                        .as_ref()
                        .unwrap()
                        .candidate
                        .as_ref()
                        .map(|c| ([c.root, c.child], c.diagnostics.log_reverse_forward))
                }
                ProposalKind::DefensiveIndependent => {
                    row.defensive_proposal =
                        Some(defensive.propose(&mut proposal_rng, FRAME, state[0], state[1])?);
                    row.defensive_proposal
                        .as_ref()
                        .unwrap()
                        .candidate
                        .as_ref()
                        .map(|c| ([c.root, c.child], c.diagnostics.log_reverse_forward))
                }
            };
            row.proposal_null = candidate.is_none();
            if let Some((new, correction)) = candidate {
                row.proposed_state = Some(new);
                let d = separation(new);
                row.proposed_d = Some(d);
                row.q_correction = Some(correction);
                row.analytic_log_weight = Some(
                    config.activity
                        * (overlap(config.exclusion_radius, d)
                            - overlap(config.exclusion_radius, row.old_d)),
                );
                row.domain_valid = in_domain(config, new);
                let gate = FlexibleSubset::new(
                    tree,
                    &state,
                    &[0, 1],
                    &new,
                    config.exclusion_radius - config.core_radius,
                )?;
                ensure!(
                    gate.spectator_count() == 0,
                    "virtual frame entered physical state"
                );
                row.hard_valid = gate.hard_valid(None, [0.; 3]);
                if row.domain_valid && row.hard_valid {
                    let physical = match chain.arm {
                        Arm::Analytic => row.analytic_log_weight.unwrap(),
                        Arm::Poisson => {
                            let sampled = gate.sample(
                                &mut gate_rng,
                                config.auxiliary_intensity,
                                config.activity,
                                config.gate.options(),
                            )?;
                            row.gate = Some(sampled);
                            sampled.log_weight
                        }
                        Arm::SingletonPath => {
                            let path = SingletonPath::new(
                                tree,
                                &state,
                                &[0, 1],
                                &new,
                                config.exclusion_radius - config.core_radius,
                            )?;
                            let sampled = path.sample(
                                &mut gate_rng,
                                config.auxiliary_intensity,
                                config.activity,
                                config.gate.options(),
                            )?;
                            let physical = sampled.aggregate.log_weight;
                            row.gate = Some(sampled.aggregate);
                            row.path_gate = Some(sampled);
                            physical
                        }
                    };
                    let log_ratio = correction + physical;
                    ensure!(log_ratio.is_finite(), "nonfinite composed MH factor");
                    row.log_ratio = Some(log_ratio);
                    row.accepted = row.log_uniform < log_ratio.min(0.);
                    if row.accepted {
                        row.mh_state = new;
                    }
                }
            }
            row.state = row.mh_state;
            // Sphere orientations are independent Haar conditionals. Updating
            // WORLD quaternions preserves both physical centers exactly.
            for body in &mut row.state {
                body.orientation = haar(&mut refresh_rng)?;
                body.validate()?;
                row.orientation_refresh_count += 1;
            }
            ensure!(
                in_domain(config, row.state),
                "accepted state left toy domain"
            );
            row.d = separation(row.state);
            Ok(())
        })();
        if let Err(error) = &result {
            row.error = Some(format!("{error:#}"));
        }
        row.attempt_cpu_seconds = cpu_seconds() - attempt_started;
        row.cpu_seconds = cpu_seconds() - started;
        ensure!(
            row.attempt_cpu_seconds.is_finite()
                && row.attempt_cpu_seconds >= 0.
                && row.cpu_seconds.is_finite()
                && row.cpu_seconds >= 0.,
            "invalid CPU clock"
        );
        counts.attempts += 1;
        counts.accepted += usize::from(row.accepted);
        counts.proposal_null += usize::from(row.proposal_null);
        counts.domain_rejected += usize::from(!row.proposal_null && !row.domain_valid);
        counts.hard_rejected += usize::from(!row.proposal_null && !row.hard_valid);
        counts.orientation_refresh_count += row.orientation_refresh_count;
        counts.errors += usize::from(row.error.is_some());
        if let Some(gate) = row.gate {
            counts.gate_calls += 1;
            counts.raw_points += gate.raw_points;
            counts.retained_points += gate.retained_points;
        }
        write_line(writer, &row)?;
        result
            .with_context(|| format!("chain {} attempt {step}; failure row retained", chain.id))?;
        state = row.state;
    }
    ensure!(
        counts.attempts == config.burn_in + config.production_steps,
        "incomplete chain"
    );
    Ok(json!({"chain":chain,"counts":counts,"final_state":state,
        "cpu_seconds":cpu_seconds()-started}))
}

fn execute(args: &Args, config: &Config, config_bytes: &[u8]) -> Result<()> {
    let shape = sphere(config.core_radius);
    let shape_sha = hash_bytes(&serde_json::to_vec(&shape)?);
    let atlas = atlas(&shape_sha);
    let model = FrozenRelativePoseProposal::from_json_str_open(
        &atlas.to_string(),
        [80.; 3],
        0.1,
        &shape_sha,
    )?;
    let docking = DockingProposal::new(
        model,
        DockingMethod::PosteriorInvolution,
        config.rho,
        [0.; 3],
    )?;
    let tree = SphereTree::new(shape.clone())?;
    let executable = std::env::current_exe()?;
    let stream_manifest: Vec<_> = config
        .chains
        .iter()
        .map(|chain| {
            json!({
                "chain_id":chain.id,"seed":chain.seed,
                "streams":(["proposal","gate","mh","orientation_refresh"].map(|label|
                    json!({"label":label,"seed_bytes":stream_seed(chain.seed,label)}))
                )
            })
        })
        .collect();
    save(
        &args.out.join("manifest.json"),
        &json!({
            "schema":1,"protocol":config,"config_sha256":hash_bytes(config_bytes),
            "executable":executable,"executable_sha256":hash_file(&executable)?,
            "example_source_sha256":hash_bytes(include_bytes!("dimer_tree_equilibrium.rs")),
            "shape":shape,"shape_sha256":shape_sha,"atlas":atlas,
            "atlas_sha256":hash_bytes(&serde_json::to_vec(&atlas)?),
            "virtual_frame":FRAME,"physical_particles":2,"tree_order":[0,1],
            "streams":stream_manifest,"rng_domain":"dimer-tree-equilibrium-v1",
            "initial_states":config.chains.iter().map(|chain|json!({"chain_id":chain.id,
                "state":initial_state(chain.initialization)})).collect::<Vec<_>>(),
            "allocation_attempts":(config.burn_in+config.production_steps)*config.chains.len(),
            "orientation_refresh":"two independent normalized four-normal world quaternions after MH, at fixed world centers",
            "scope":"bounded toy equilibrium control; no production kernel, proteins, adaptation, continuation or native labels"
        }),
    )?;
    let started = cpu_seconds();
    let mut summaries = Vec::new();
    for chain in &config.chains {
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(args.out.join(format!("{}.jsonl", chain.id)))?,
        );
        save(
            &args.out.join("status.json"),
            &json!({"phase":"running",
            "chain_id":chain.id,"completed_chains":summaries.len()}),
        )?;
        let summary = run_chain(config, chain, &tree, &docking, &mut writer)?;
        eprintln!(
            "completed {}: {} attempts, {:.3} CPU seconds",
            chain.id,
            config.burn_in + config.production_steps,
            summary["cpu_seconds"].as_f64().unwrap()
        );
        summaries.push(summary);
        save(
            &args.out.join("summary.json"),
            &json!({"schema":1,
            "chains":summaries,"cpu_seconds":cpu_seconds()-started}),
        )?;
    }
    save(
        &args.out.join("status.json"),
        &json!({"phase":"completed",
        "completed_chains":summaries.len(),"attempts":
        (config.burn_in+config.production_steps)*config.chains.len(),
        "cpu_seconds":cpu_seconds()-started}),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bytes = fs::read(&args.config)?;
    let config: Config = serde_json::from_slice(&bytes)?;
    config.validate()?;
    fs::create_dir(&args.out).context("output must be a new directory with an existing parent")?;
    let result = execute(&args, &config, &bytes);
    if let Err(error) = &result {
        // Never relabel a failed control complete or retry/discard its draws.
        save(
            &args.out.join("status.json"),
            &json!({"phase":"failed",
            "error":format!("{error:#}")}),
        )?;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            schema: 1,
            core_radius: 0.2,
            exclusion_radius: 1.,
            activity: 1.5,
            auxiliary_intensity: 24.,
            root_radius: 2.,
            min_separation: 0.4,
            max_separation: 2.5,
            rho: 0.7,
            uniform_probability: 0.5,
            uniform_half_width: 2.5,
            gate: Gate {
                max_cells: 255,
                max_depth: 5,
                min_width: 0.,
            },
            burn_in: 0,
            production_steps: 1,
            chains: vec![Chain {
                id: "control".into(),
                arm: Arm::Analytic,
                initialization: Initialization::Contact,
                seed: 123,
                proposal_kind: ProposalKind::Tree,
            }],
        }
    }

    #[test]
    fn frozen_contract_rejects_invalid_and_duplicate_allocations() {
        config().validate().unwrap();
        for mutate in [
            |c: &mut Config| c.chains.push(c.chains[0].clone()),
            |c: &mut Config| c.production_steps = 0,
            |c: &mut Config| c.min_separation = 0.3,
            |c: &mut Config| c.rho = 1.1,
            |c: &mut Config| c.auxiliary_intensity = 0.,
            |c: &mut Config| c.chains[0].id = "../bad".into(),
        ] {
            let mut c = config();
            mutate(&mut c);
            assert!(c.validate().is_err());
        }
        let mut c = config();
        let mut other = c.chains[0].clone();
        other.id = "other".into();
        c.chains.push(other);
        assert!(c.validate().is_err(), "seeds must also be independent");
    }

    #[test]
    fn reference_domain_has_no_child_wall_and_orientation_refresh_moves_no_centers() {
        let c = config();
        let mut state = initial_state(Initialization::Dispersed);
        assert!(norm(state[1].position) > c.root_radius);
        assert!(in_domain(&c, state));
        let centers = state.map(|p| p.position);
        let mut rng = stream(3, "orientation_refresh");
        for p in &mut state {
            p.orientation = haar(&mut rng).unwrap();
            p.validate().unwrap();
        }
        assert_eq!(state.map(|p| p.position), centers);
        assert_eq!(overlap(1., 2.), 0.);
        assert!((overlap(1., 0.) - 4. * PI / 3.).abs() < 1e-14);
    }

    #[test]
    fn streams_and_four_chart_atlas_are_deterministic_and_distinct() {
        let labels = ["proposal", "gate", "mh", "orientation_refresh"];
        let seeds: BTreeSet<_> = labels.map(|s| stream_seed(123, s)).into_iter().collect();
        assert_eq!(seeds.len(), 4);
        assert_ne!(stream_seed(123, "proposal"), stream_seed(124, "proposal"));
        let sha = hash_bytes(&serde_json::to_vec(&sphere(0.2)).unwrap());
        let raw = atlas(&sha);
        let model =
            FrozenRelativePoseProposal::from_json_str_open(&raw.to_string(), [80.; 3], 0.1, &sha)
                .unwrap();
        for axis in 0..4 {
            let q = std::array::from_fn(|j| if j == axis { 1. } else { 0. });
            assert!(
                model
                    .relative_log_density([0.; 3], rotation(q))
                    .unwrap()
                    .is_finite()
            );
        }
    }
}
